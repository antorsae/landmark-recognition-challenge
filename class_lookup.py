import commands
import sys
import jpeg4py as jpeg
from PIL import Image
import cv2
import numpy as np
label = sys.argv[1]

test_id = None
if ',' in label:
    test_id, label = label.split(',')

_, output = commands.getstatusoutput(
    "grep -E ',%s$' ../../../train.csv | cut -f1 -d','" % label)


def read_image(image_id, test):
    if test == 1:
        img_path = '../../test-dl/%s.jpg' % image_id
    else:
        img_path = '../../train-dl/%s.jpg' % image_id
    try:
        img = jpeg.JPEG(img_path).decode()
    except:
        img = np.array(Image.open(img_path))
    img = cv2.resize(img, (512, 512))
    return img


i = 0
for image_id in output.split('\n'):
    img = read_image(image_id, 0)
    if len(img.shape) != 3:
        continue
    img = img[:, :, ::-1]
    cv2.imwrite('/tmp/img%d.jpg' % i, img)
    i = i + 1
    if i > 3:
        break

commands.getstatusoutput('montage -geometry +2+2 ' + ' '.join(
    ['/tmp/img%d.jpg' % j for j in range(i)]) + ' /tmp/%s.jpg' % label)

_, output = commands.getstatusoutput(
    'aws s3 cp /tmp/%s.jpg s3://kaggleglm/class_lookup/ --grants full=uri=http://acs.amazonaws.com/groups/global/AllUsers'
    % label)
print 'https://s3-us-west-2.amazonaws.com/kaggleglm/class_lookup/%s.jpg' % label

if test_id is not None:
    img = read_image(test_id, 1)
    img = img[:, :, ::-1]
    cv2.imwrite('/tmp/%s.jpg' % test_id, img)

    commands.getstatusoutput('montage -label %t ' + '-geometry +2+2  /tmp/%s.jpg /tmp/%s.jpg /tmp/%s_%s.jpg' % (test_id, label, test_id, label))

    _, output = commands.getstatusoutput(
        'aws s3 cp /tmp/%s_%s.jpg s3://kaggleglm/class_lookup/ --grants full=uri=http://acs.amazonaws.com/groups/global/AllUsers'
        % (test_id, label))
    print 'https://s3-us-west-2.amazonaws.com/kaggleglm/class_lookup/%s_%s.jpg' % (test_id, label)
