"""Create small synthetic demo datasets; no real user data is modified."""
import argparse
import csv
from pathlib import Path
import math
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/zoo_demo')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(5)

    def write_csv(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    rows = []
    for i in range(48):
        a, b, c = rng.uniform(-1,1), rng.uniform(-1,1), rng.uniform(-1,1)
        rows.append([a,b,c, 2*a-b+0.3*c])
    write_csv(root/'tabular_regression.csv', ['a','b','c','target'], rows)
    write_csv(root/'tabular_classification.csv', ['a','b','c','class'],
              [[a,b,c, int(a+b > 0)] for a,b,c,_ in rows])
    write_csv(root/'text.csv', ['text','label'],
              [[s,lab] for s,lab in [
                  ('this is fantastic', 'positive'),('very nice', 'positive'),('excellent work', 'positive'),
                  ('I am happy', 'positive'),('good experience', 'positive'),('amazing day', 'positive'),
                  ('very bad', 'negative'),('I dislike this', 'negative'),('awful experience', 'negative'),
                  ('bad results', 'negative'),('terrible situation', 'negative'),('poor product', 'negative')]])
    with (root/'timeseries.csv').open('w', encoding='utf-8') as f:
        f.write('value\n')
        for i in range(96): f.write(f'{math.sin(i/8):.6f}\n')
    (root/'lm.txt').write_text(('The small transformer learns character patterns in this example. '\
                               'This is not a pretrained large language model. ')*12, encoding='utf-8')

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print('Install Pillow to generate demo image and segmentation datasets')
        return
    for category in ['red','blue']:
        for i in range(8):
            p = root/'images'/category/f'{category}_{i:02}.png'
            p.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new('RGB', (64,64), (255,60+i*3,60) if category == 'red' else (40,60,255-i*3))
            image.save(p)
            # Synthetic spectrogram image classification demo
            a = root/'spectrograms'/category/f'{category}_{i:02}.png'
            a.parent.mkdir(parents=True, exist_ok=True)
            image.save(a)
    for i in range(8):
        rgb = Image.new('RGB', (64,64), (25,25,25))
        mask = Image.new('L', (64,64), 0)
        draw = ImageDraw.Draw(rgb)
        dmask = ImageDraw.Draw(mask)
        box = (10+i, 10, 28+i, 30)
        draw.rectangle(box, fill=(240,100,80))
        dmask.rectangle(box, fill=255)
        a = root/'segmentation'/'images'/f'{i:02}.png'
        b = root/'segmentation'/'masks'/f'{i:02}.png'
        a.parent.mkdir(parents=True, exist_ok=True)
        b.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(a)
        mask.save(b)
    print('Demo datasets created in:', root.resolve())
    print('Use e.g. data/zoo_demo/images or data/zoo_demo/text.csv on /model-zoo')

if __name__ == '__main__':
    main()
