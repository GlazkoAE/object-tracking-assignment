import asyncio
import glob
from collections import Counter

import numpy as np
from fastapi import FastAPI, WebSocket

from soft_tracker.sort import Sort
from track_5 import country_balls_amount, track_data

# TODO: Metrics calculation

app = FastAPI(title="Tracker assignment")
imgs = glob.glob("imgs/*")
country_balls = [
    {"cb_id": x, "img": imgs[x % len(imgs)]} for x in range(country_balls_amount)
]
tracker = Sort(max_age=10, hit_sum=1)
print("Started")


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """

    detections = np.empty((1, 4), dtype=int)
    for obj in el["data"]:
        bbox = obj["bounding_box"]
        if len(bbox) > 0:
            bbox = np.array(bbox)
            bbox = np.reshape(bbox, (1, 4))
            detections = np.concatenate((detections, bbox), axis=0)

    detections = detections[1:, :]
    out = tracker.update(detections)
    for i in range(len(out)):
        el["data"][i]["track_id"] = out[i][-1]

    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    return el


def get_metric(track_ids: dict):
    metrics_list = []

    for _, val in track_ids.items():
        if len(val) == 0:
            metrics_list.append(0)
        else:
            track_id = Counter(val).most_common(1)[0]
            if track_id[0] is None:
                track_id = Counter(val).most_common(2)[1]
            metrics_list.append(track_id[1] / len(val))
    metric = sum(metrics_list) / len(metrics_list)
    return metric


def update_track_ids(el: dict, track_ids: dict):
    for obj in el["data"]:
        cb_id = obj["cb_id"]
        track_id = obj["track_id"]
        track_ids[cb_id].append(track_id)
    return track_ids


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Accepting client connection...")
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    track_ids_soft = {i: [] for i in range(country_balls_amount)}
    track_ids_strong = {i: [] for i in range(country_balls_amount)}
    for el in track_data:
        await asyncio.sleep(0.5)
        el_soft = tracker_soft(el)
        track_ids_soft = update_track_ids(el_soft, track_ids_soft)
        # TODO: part 2
        # el_strong = tracker_strong(el)
        # track_ids_strong = update_track_ids(el_strong, track_ids_strong)
        # отправка информации по фрейму
        await websocket.send_json(el_soft)
    metric_soft = get_metric(track_ids_soft)
    metric_strong = get_metric(track_ids_strong)
    print(f"metric_soft: {metric_soft}")
    print(f"metric_strong: {metric_strong}")
    print("Bye..")
