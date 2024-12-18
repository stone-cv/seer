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
from shared_db_models.database import SessionLocal
from shared_db_models.models.models import VideoFile


# start_time = datetime.fromisoformat("2024-02-06T16:00:00Z".replace("Z", "+00:00"))
# end_time = datetime.fromisoformat("2024-02-06T17:59:59Z".replace("Z", "+00:00"))


# TODO: check for free space

def unix_time_from_file(
        file_time: str
) -> int:
    """
    Функция, конвертирующая время в unix-формат.

    Args:
        file_time (str): строка со временем в формате ISO

    Returns:
        int: время в unix-формате
    """
    datetime_obj = datetime.fromisoformat(file_time.replace("Z", "+03:00"))  # Moscow timezone. TODO rm hardcode
    return int(datetime_obj.timestamp())


async def get_files_list(
    channel: int,
    recorder_ip: str,
    start_time: datetime,
    end_time: datetime
) -> dict:
    """
    Функция, позволяющая найти на видеорегистраторе файлы в заданном диапазоне времени

    Args:
        channel (int): ID канала (камеры) в формате TrackID
        recorder_ip (str): IP-адрес видеорегистратора
        start_time (datetime): начало запрашиваемого периода
        end_time (datetime): окончание запрашиваемого периода
    
    Returns:
        files_dict (dict): словарь с полученными данными о видеофайлах
    """

    files_dict = {}

    retry_count = 0
    success = False

    search_url = 'ContentMgmt/search'
    api_url = f'http://{recorder_ip}/ISAPI/{search_url}'

    async with httpx.AsyncClient() as client:
        while retry_count <= 10 and not success:
            try:
                # отправка запроса с заданными параметрами
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

                else:
                    success = True

            # except (httpx.TimeoutException, httpx.ReadTimeout, asyncio.CancelledError) as exc:
            except Exception as exc:
                retry_count += 1
                logger.error(f"Videofile retrieval task error, retry count: {retry_count}\n{exc} {traceback.format_exc()}")
                    
                await asyncio.sleep(5)
    
    if success:
        # TODO: Add 401 response handle

        # парсинг ответа с данными о видеофайлах
        root = etree.fromstring(response.text.replace('<?xml version="1.0" encoding="UTF-8" ?>\n', ''))
        logger.debug(root)
        items = []

        if len(root.getchildren()) > 4:
            video_list = root.getchildren()[4].getchildren()[:-1]  # не берем последний элемент, т.к. этот видеофайл еще в процессе записи
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

        # формируем словарь {номер канала: [данные с регистратора]}
        files_dict[channel] = items
        logger.debug(f'{len(files_dict[channel])} file(s) retrieved for download: {files_dict}')

        return files_dict


async def download_files(
        #channel: int,
        recorder_ip: str,
        file_id: int,
        data: dict
) -> VideoFile:
    """
    Функция, скачивающая файлы с видеорегистратора

    Args:
        recorder_ip (str): IP-адрес видеорегистратора
        file_id (int): ID видеофайла в БД
        data (dict): словарь с данными о видеофайле
    
    Returns:
        videofile (VideoFile): объект видеофайла
    """
    # название папки для хранения файлов (== дата видеозаписи)
    date_folder_name = f"{data['startTime'][:10]}"

    # название сохраняемого файла в формате:
    # <номер канала>_<время начала видеофрагмента>_<время окончания видеофрагмента>.<расширение>
    file_name = f"{data['trackID']}_{unix_time_from_file(data['startTime'])}_{unix_time_from_file(data['endTime'])}.mp4"

    reg_path = f'{cfg.download_dir}/{date_folder_name}'

    if not os.path.exists(reg_path):
        os.makedirs(reg_path, exist_ok=True)

    data_filepath = os.path.join(reg_path, file_name)

    download_start = datetime.now()

    # параметры загрузки файлов
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
                # отправка запроса на скачивание видеофайла
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

                    # сохранение скачиваемого видео в файл
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
        # обновляем запись о видеофайле в БД
        async with SessionLocal() as session:
            videofile = await VideoFile.update(
                    db_session=session,
                    id=file_id,
                    path=data_filepath,
                    download_start=download_start,
                    download_end=datetime.now(),
                    is_downloaded=True
                )

        return videofile


# if __name__ == "__main__":
    # asyncio.run(ffmpeg_download())
