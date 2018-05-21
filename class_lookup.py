import commands
import sys
import jpeg4py as jpeg
from PIL import Image
import cv2
label = sys.argv[1]

_, output = commands.getstatusoutput(
    "grep -E ',%s$' ../../../train.csv | cut -f1 -d','" % label)


def read_image(image_id):
    img_path = '../../train-dl/%s.jpg' % image_id
    try:
        img = jpeg.JPEG(img_path).decode()
    except:
        img = np.array(Image.open(img_path))
    img = cv2.resize(img, (512, 512))
    return img


i = 0
for image_id in output.split('\n'):
    img = read_image(image_id)
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
