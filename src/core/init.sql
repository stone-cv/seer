CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    type_id INT,
    camera_id INT,
    time TIMESTAMP,
    machine VARCHAR(255),
    stone_number INT,
    stone_area VARCHAR(255),
    comment VARCHAR(255),
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE event_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    url VARCHAR(255),
    roi VARCHAR(255),
    deleted BOOLEAN DEFAULT FALSE
);

INSERT INTO cameras (id, name, url, roi, deleted) 
VALUES (1, 'TR-D3152ZIR2V2', 'rtsp://login:password@192.168.0.52:554/live/main', '((624, 7), (1539, 555))', false);

INSERT INTO event_types (id, name, deleted) 
VALUES 
(1, 'Новый товарный блок на станке', false),
(2, 'Товарный блок убран со станка', false),
(3, 'Начало распила товарного блока', false),
(4, 'Окончание распила товарного блока', false),
(5, 'Вычислена площадь камня', false);
