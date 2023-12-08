import time
import multiprocessing as mp

from main import extract_frame

class Detector:
    def __init__(self, args):
        """
        Инициализация класса детектора. В котором мы создаем все необходимые переменные, получаем данные о них.
        :param args:
        """
        # self.weight = args.weights  # храниться путь до весов
        # self.weights_file = None
        # self.cfg_file = None
        # self.names_file = None

        # self.force_exit = mp.Event()

        # self.id_camera = args.id_camera  # храниться идентификатор камеры для которой мы обрабатываем камеру
        # self.config_path = args.config_path  # хранится путь до конфигурационного файла подключения к БД
        # self.video_path = args.file_path  # хранится путь до видео
        # self.need_stop = False
        self.frame_queue = mp.Queue(maxsize=50)
        self.det_queue = mp.Queue()
        self.track_queue = mp.Queue()

        # self.database = None
        # self.in_out_zones = []
        # self.video = Reader(self.video_path)

        # self.video_reader, self.video_reader_need_exit = None, None
        # self.preprocess, self.preprocess_need_exit = None, None
        # self.postprocess, self.post_process_need_exit = None, None
        # self.wait_save = mp.Event()
        # self.save_thread, self.save_thread_stop = None, None
        # # self.post_process, self.post_process_need_exit =
        # self.wait_end = False

    async def fill_frame(self):
        bt = time.time()
        for frame in extract_frame(video_path='static/baggage-on-belt.mov', fps=5):
            frame_rgb = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)

            frame_h = frame_rgb.shape[0]
            frame_w = frame_rgb.shape[1]

            frame_resized = cv2.resize(frame_rgb, (416, 416),
                                       interpolation=cv2.INTER_LINEAR)
            frame.frame = frame_resized
            # img_for_detect = detector.make_image(412, 412, 3)
            self.frame_queue.put(frame)

        # self.wait_end = True
        return
