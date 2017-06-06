import subprocess
import time
import glob
from PIL import Image
from PIL.ExifTags import TAGS
from PIL import ImageStat
import pandas as pd

def get_exif(fn):
    ret = {}
    try:
        i = Image.open(fn)
        w, h = i.size
        ret['megapixels'] = str(w*h)
        info = i._getexif()
        if (info != None):
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                ret[str(decoded)] = str(value)
    except:
        return None
    return ret

t1 = time.time()

FILES = glob.glob('./photos/*.jpg')

BAD = ['Copyright', 'YCbCrCoefficients', 'UserComment', 'LensModel', 'PrimaryChromaticities', 'ImageDescription', 'MakerNote', 'PrintImageMatching', 'GPSInfo', 'XPAuthor']

count = 0
bad = 0
edata = []
rare = []
df = pd.DataFrame()
for f in FILES:#[:2000]:
    out = subprocess.check_output(["md5sum", f])
    md5sum, filename = out[:-1].decode('utf-8').split()
    exf = get_exif(f)
    if (exf != None):
        if (len(exf) > 0):
            count += 1
            for b in BAD:
                exf.pop(b, None)
            exf['my_filename'] = f
            exf['my_md5sum'] = md5sum
            d2 = pd.DataFrame(exf, index=[0])
            df = df.append(d2, ignore_index=True)
            for k, v in exf.items():
                if (type(k) == str):
                    if (len(str(v)) > 50):
                        rare.append(k)
                    else:
                        edata.append(k)
    else:
        bad += 1

df.to_csv('exif_data.csv')
t2 = time.time()

print("Processed files: {}/{} = {}".format(count, len(FILES), count*1.0/len(FILES)))
print("Bad files: {}/{} = {}".format(bad, len(FILES), bad*1.0/len(FILES)))
print("Time {} s".format(t2-t1))
