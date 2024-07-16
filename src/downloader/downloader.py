import os
import asyncio
import httpx
import traceback
import aiofiles

from datetime import datetime
from time import perf_counter
from tqdm import tqdm
from lxml import etree
from sqlalchemy.ext.asyncio.session import AsyncSession

import core.config as cfg
from core.logger import logger
from core.utils import xml_helper


# start_time = datetime.fromisoformat("2024-02-06T16:00:00Z".replace("Z", "+00:00"))
# end_time = datetime.fromisoformat("2024-02-06T17:59:59Z".replace("Z", "+00:00"))


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

    retry_count = 0
    success = False

    search_url = 'ContentMgmt/search'
    api_url = f'http://{recorder_ip}/ISAPI/{search_url}'

    async with httpx.AsyncClient() as client:
        while retry_count <= 10 and not success:
            try:
                # перевод xml в json
                response = await client.post(
                    api_url,
                    auth=httpx.DigestAuth(username=cfg.cam_login, password=cfg.cam_password),
                    content=xml_helper(start_time, end_time, channel),
                    timeout=30
                )
                logger.debug(response)

                if response.status_code != 200:
                    retry_count += 1
                    logger.error(f'Videofile retrieval task error: response status code {response.status_code}; retry count: {retry_count}.')
                    await asyncio.sleep(5)
                    continue

                success = True

            # except (httpx.TimeoutException, httpx.ReadTimeout, asyncio.CancelledError) as exc:
            except Exception as exc:
                retry_count += 1
                logger.error(f"Videofile retrieval task error, retry count: {retry_count}\n{exc} {traceback.format_exc()}")
                    
                await asyncio.sleep(5)
    
    if success:
        # TODO: Add 401 response handle
        root = etree.fromstring(response.text.replace('<?xml version="1.0" encoding="UTF-8" ?>\n', ''))
        logger.debug(root)
        items = []

        if len(root.getchildren()) > 4:
            video_list = root.getchildren()[4].getchildren()[:-1]  # the last file is still being recorded
            for match in video_list:
                data = {}
                for el in match.getchildren():
                    del_url = [
                        '{http://www.hikvision.com/ver20/XMLSchema}',
                        '{http://www.std-cgi.com/ver20/XMLSchema}',
                        '{http://www.isapi.org/ver20/XMLSchema}',
                    ]

                    for url in del_url:
                        if el.text != '\n':
                            dict_key = el.tag.replace(url, '') if url in el.tag else el.tag
                            data[dict_key] = el.text
                        else:
                            for el_ch in el.getchildren():
                                dict_key = el_ch.tag.replace(url, '')if url in el_ch.tag else el_ch.tag
                                data[dict_key] = el_ch.text
                if data['startTime'] == data['endTime']:
                    continue
                items.append(data)

        files_dict[channel] = items  # Формируем словарь {номер канала: [данные с регистратора]}
        logger.debug(f'{len(files_dict[channel])} file(s) retrieved for download: {files_dict}')

        return files_dict


async def download_files(
        channel: int,
        recorder_ip: str,
        data: dict
) -> None:

    # for data in files_dict[channel]:

    date_folder_name = f"{data['startTime'][:10]}"  # redo
    file_name = f"{data['trackID']}_{unix_time_from_file(data['startTime'])}_{unix_time_from_file(data['endTime'])}.mp4"

    reg_path = f'{cfg.download_dir}/{date_folder_name}'

    if not os.path.exists(reg_path):
        os.makedirs(reg_path, exist_ok=True)

    data_filepath = os.path.join(reg_path, file_name)

    # Начинаем загрузку файлов
    api_url = f'http://{recorder_ip}/ISAPI/'
    download_url = api_url + 'ContentMgmt/download'
    playback_uri = data['playbackURI']
    download_xml = f'<downloadRequest version="1.0" xmlns="http://www.isapi.org/ver20/XMLSchema">' \
                    f'<playbackURI>{playback_uri}</playbackURI></downloadRequest>'

    retry_count = 0
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
                    timeout=60
                ) as response:
                    logger.debug(f"Download task: response status code {response.status_code}")
                    logger.debug(response)

                    if response.status_code != 200:
                        retry_count += 1
                        logger.error(f'Download task error: response status code {response.status_code}; retry count: {retry_count}.')
                        logger.debug(response)
                        await asyncio.sleep(5)
                        continue

                    total_size_in_bytes = int(response.headers.get('content-length', 0))

                    logger.debug(f"File size: {total_size_in_bytes} bytes")

                    bt = perf_counter()

                    async with aiofiles.open(data_filepath, 'wb') as video_file:
                        progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)

                        async for chunk in response.aiter_bytes():
                            await video_file.write(chunk)
                            progress_bar.update(len(chunk))

                        progress_bar.close()

                    et = perf_counter() - bt
                    dw = (total_size_in_bytes / (datetime.now().timestamp() - unix_time_from_file(data['startTime']))) / 1024 / 1024 * 8
                    logger.debug(f"File {file_name}; time {et} s; speed {dw} mb/s")

                    success = True
                    logger.info(f"File {file_name} downloaded")

            except (httpx.TimeoutException, httpx.ReadTimeout, asyncio.CancelledError) as exc:
                retry_count += 1
                logger.error(f"Download task error\nFile: {file_name}, retry count: {retry_count}\n{exc} {traceback.format_exc()}")

            await asyncio.sleep(5)
    
    if success:
        return data_filepath


# if __name__ == "__main__":
    # asyncio.run(ffmpeg_download())
