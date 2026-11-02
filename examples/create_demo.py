"""Original deterministic test artwork; not a benchmark or dataset sample."""
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    image = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((52, 172, 212, 332), fill=(220, 35, 35))
    draw.ellipse((294, 172, 454, 332), fill=(20, 75, 225))
    path = Path(__file__).with_name('demo.png')
    image.save(path)
    print(f'Created {path.name}: red square on the left, blue circle on the right.')


if __name__ == '__main__':
    main()
