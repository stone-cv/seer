CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    track_id INTEGER,
    url VARCHAR(255),
    roi VARCHAR(255),
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE event_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    type_id INTEGER REFERENCES event_types(id),
    camera_id INTEGER REFERENCES cameras(id),
    time TIMESTAMP,
    machine VARCHAR(255),
    stone_number INTEGER,
    stone_area VARCHAR(50),
    comment VARCHAR(255),
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE video_files (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    path VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    orig_name VARCHAR(255),
    playback_uri VARCHAR(255),
    vid_start TIMESTAMP,
    vid_end TIMESTAMP,
    is_downloaded BOOLEAN DEFAULT FALSE,
    download_start TIMESTAMP,
    download_end TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    det_start TIMESTAMP,
    det_end TIMESTAMP,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE daily_cam_check (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER REFERENCES cameras(id),
    date TIMESTAMP,
    is_processed BOOLEAN,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE daily_all_cam_check (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP,
    is_processed BOOLEAN,
    deleted BOOLEAN DEFAULT FALSE
);

INSERT INTO cameras (id, name, track_id, url, roi, deleted) 
VALUES (1, 'TR-D3152ZIR2V2', 201, 'rtsp://login:password@192.168.0.52:554/live/main', '((624, 7), (1539, 555))', false);

INSERT INTO event_types (id, name, deleted) 
VALUES 
(1, 'Новый товарный блок на станке', false),
(2, 'Товарный блок убран со станка', false),
(3, 'Начало распила товарного блока', false),
(4, 'Окончание распила товарного блока', false),
(5, 'Вычислена площадь камня', false);
