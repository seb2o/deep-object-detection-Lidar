import ffmpeg


def main():
    ffmpeg.input(
        'frame_%06d.PNG',
        start_number=1505,
        framerate=5
    ).output(
        'vid.mp4',
        pix_fmt='yuv420p'
    ).run()


if __name__ == '__main__':
    main()
