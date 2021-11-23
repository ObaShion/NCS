import pyaudio
import numpy as np
import cv2

SAMPLE_RATE = 44100
FRAME_SIZE = 2048
INT16_MAX = 32767
SAMPLING_SIZE = FRAME_SIZE * 4
WIDTH = 800     # 表示領域の幅
HEIGHT = 600    # 表示領域の高さ

spectram_range = [int(22050 / 2 ** (i/10)) for i in range(100, -1,-1)]
freq = np.abs(np.fft.fftfreq(SAMPLING_SIZE, d=(1/SAMPLE_RATE)))
spectram_array = (freq <= spectram_range[0]).reshape(1,-1)
for index in range(1, len(spectram_range)):
    tmp_freq = ((freq > spectram_range[index - 1]) & (freq <= spectram_range[index])).reshape(1,-1)
    spectram_array = np.append(spectram_array, tmp_freq, axis=0)


part_w = WIDTH / len(spectram_range)
part_h = HEIGHT / 100
img = np.full((HEIGHT, WIDTH, 3), 0, dtype=np.uint8)


audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                    input=True, input_device_index=1, frames_per_buffer=FRAME_SIZE)

sampling_data = np.zeros(SAMPLING_SIZE)
while True:
    frame = stream.read(FRAME_SIZE)
    frame_data = np.frombuffer(frame, dtype="int16") / INT16_MAX
    sampling_data = np.concatenate([sampling_data, frame_data])
    if sampling_data.shape[0] > SAMPLING_SIZE:
        sampling_data = sampling_data[sampling_data.shape[0] - SAMPLING_SIZE:]

    fft = np.abs(np.fft.fft(sampling_data))

    spectram_data = np.dot(spectram_array, fft)

    cv2.rectangle(img, (0,0), (WIDTH, HEIGHT), (0,0,0), thickness=-1)
    for index, value in enumerate(spectram_data):
        rad = (2 * np.pi) * (index / len(spectram_data))
        x1 = int(WIDTH / 2 + np.sin(rad) * 80)
        y1 = int(HEIGHT / 2 - np.cos(rad) * 80)
        rad = (2 * np.pi) * (index / len(spectram_data))
        x2 = int(WIDTH / 2 + np.sin(rad) * (80 + value/4))
        y2 = int(HEIGHT / 2 - np.cos(rad) * (80 + value/4))
        cv2.line(img, (x1, y1), (x2, y2), (0, 225, 0), thickness=2)
    cv2.namedWindow("          ", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("NCS", img)

    # 終了キーチェック
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 0x1b:
        break

# マイク サンプリング終了処理
stream.stop_stream()
stream.close()
audio.terminate()