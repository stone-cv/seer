import os
import asyncio
import time  #?
import httpx
import traceback
import aiofiles
from datetime import datetime

from lxml import etree
from sqlalchemy.ext.asyncio.session import AsyncSession

import core.config as cfg
from core.logger import logger
from core.utils import xml_helper


start_time = datetime.fromisoformat("2024-02-06T16:00:00Z".replace("Z", "+00:00"))  # timezone ?
end_time = datetime.fromisoformat("2024-02-06T17:59:59Z".replace("Z", "+00:00"))  # timezone ?


# TODO: check for free space

def unix_time_from_file(file_time: str) -> int:  # copied
    datetime_obj = datetime.fromisoformat(file_time.replace("Z", "+03:00"))  # Moscow timezone
    return int(datetime_obj.timestamp())


async def get_files_list(
    channel: int,
    recorder_ip: str,
    start_time: datetime,
    end_time: datetime
) -> dict:
    """

    :param channel: Номер канала в формате TrackID.
    :param start_time: Дата начала периода запроса в формате datetime.
    :param end_time: Дата окончания периода запроса в формате datetime.
    :param  recorder
    :return: Словарь с данными с видеорегистратора.
    """

    files_dict = {}

    search_url = 'ContentMgmt/search'
    api_url = f'http://{recorder_ip}/ISAPI/{search_url}'

    # перевод xml в json
    async with httpx.AsyncClient() as client:
        response = await client.post(
            api_url,
            auth=httpx.DigestAuth(username=cfg.cam_login, password=cfg.cam_password),
            content=xml_helper(start_time, end_time, channel),
            timeout=30
        )
        logger.debug(response)

    # TODO: Add 401 response handle, because we can fucked this up if login/password is incorrect 
    root = etree.fromstring(response.text.replace('<?xml version="1.0" encoding="UTF-8" ?>\n', ''))
    logger.debug(root)
    items = []

    if len(root.getchildren()) > 4:
        video_list = root.getchildren()[4].getchildren()[:-1]  # why -1 ?
        # video_list = root.getchildren()[4].getchildren()
        for match in video_list:
            d = {}
            for el in match.getchildren():
                del_url = [
                    '{http://www.hikvision.com/ver20/XMLSchema}',
                    '{http://www.std-cgi.com/ver20/XMLSchema}',
                    '{http://www.isapi.org/ver20/XMLSchema}',
                ]

                for url in del_url:
                    if el.text != '\n':
                        dict_key = el.tag.replace(url, '') if url in el.tag else el.tag
                        d[dict_key] = el.text
                    else:
                        for el_ch in el.getchildren():
                            dict_key = el_ch.tag.replace(url, '')if url in el_ch.tag else el_ch.tag
                            d[dict_key] = el_ch.text
            if d['startTime'] == d['endTime']:
                continue
            items.append(d)
    files_dict[channel] = items  # Формируем словарь {номер канала: [данные с регистратора]}
    logger.info(f'{len(files_dict[channel])} files retrieved for download')
    logger.debug(f'Files retrieved for download: {files_dict}')

    return files_dict


async def download_files(
        channel: int,
        recorder_ip: str,
        files_dict: dict
) -> None:

    for data in files_dict[channel]:
        date_folder_name = f"{data['startTime'][:10]}"  # redo
        # file_name = f"{data['trackID']}_" + data['playbackURI'].split('&')[-2].replace('name=', '') + \
        #         f"_{unix_time_from_file(data['startTime'])}_{unix_time_from_file(data['endTime'])}" + ".mp4"
        file_name = f"{data['trackID']}_{unix_time_from_file(data['startTime'])}_{unix_time_from_file(data['endTime'])}.mp4"

        # reg_path = f'static/{date_folder_name}'
        reg_path = f'static/for_annotation'

        if not os.path.exists(reg_path):
            os.makedirs(reg_path, exist_ok=True)

        data_filepath = os.path.join(reg_path, file_name)

        # Начинаем загрузку файлов
        api_url = f'http://{recorder_ip}/ISAPI/'
        download_url = api_url + 'ContentMgmt/download'
        playback_uri = data['playbackURI']
        download_xml = f'<downloadRequest version="1.0" xmlns="http://www.isapi.org/ver20/XMLSchema">' \
                        f'<playbackURI>{playback_uri}</playbackURI></downloadRequest>'

        try:
            retry_count = 0
            last_status_code = 0
            success = False

            total_size_in_bytes = 0

            async with httpx.AsyncClient() as client:
                while retry_count <= 10 and not success:
                    try:
                        async with client.stream(
                            'POST',
                            download_url,
                            auth=httpx.DigestAuth(username=cfg.cam_login, password=cfg.cam_password),
                            content=download_xml,
                            timeout=None
                        ) as response:
                            logger.info(f"Download task: response {response.status_code}")

                            if response.status_code != 200:
                                last_status_code = response.status_code
                                retry_count += 1
                                logger.warn(f'Download task StatusError: Response status: {response.status_code}. Retry count: {retry_count}.')
                                await asyncio.sleep(5)
                                continue

                            total_size_in_bytes = int(response.headers.get('content-length', 0))
 
                            logger.info(f"File size in bytes {total_size_in_bytes}")

                            bt = time.perf_counter()

                            async with aiofiles.open(data_filepath, 'wb') as video_file:
                                async for chunk in response.aiter_bytes():
                                    await video_file.write(chunk)

                            et = time.perf_counter() - bt
                            dw = (total_size_in_bytes / (datetime.now().timestamp() - unix_time_from_file(data['startTime']))) / 1024 / 1024 * 8
                            logger.info(f"File {file_name}; time { et } s; speed { dw } mb/s")

                            success = True
                            logger.info(f"File {file_name} downloaded")

                    except httpx.TimeoutException as exc:
                        retry_count += 1
                        logger.error(f"Download task TimeoutError\n Data id: , retry count: {retry_count}\n {exc} {traceback.format_exc()}")

                    await asyncio.sleep(5)

        except Exception as exc:  # ?
            logger.error(exc)


if __name__ == "__main__":
    files_dict = asyncio.run(get_files_list(
        channel=cfg.channel,
        recorder_ip=cfg.recorder_ip,
        start_time=start_time,
        end_time=end_time
    ))
    asyncio.run(download_files(
        channel=cfg.channel,
        recorder_ip=cfg.recorder_ip,
        files_dict=files_dict
    ))
