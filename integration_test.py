import pytest

from main_camera_video_thread import main_camera_thread

results = main_camera_thread(
    isCamera=False, videoName="./Speaker_Detection/2.mpg", silent=True)
print(results)


def test1():
    # at first he was not speaking yet (only mouth open) - Neutral Emotion
    assert results[0] == (True, 'neutral but not speaking'), "test failed"


def test2():
    # then he was speaking - Neutral Emotion
    assert results[1] == (True, 'neutral'), "test failed"
