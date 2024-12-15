import os
import cv2

dirname = '/irsa/picmatch/tzb_code_js/match_code/IRSA_Match/works/hebei/TEST/testdata/SC_2'


for imname in os.listdir(dirname):
    img = cv2.imread(os.path.join(dirname, imname))

    spath = dirname + '_1'
    os.makedirs(spath, exist_ok=True)
    trans_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(spath, imname), trans_image)

    spath = dirname + '_2'
    os.makedirs(spath, exist_ok=True)
    trans_image = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(spath, imname), trans_image)

    spath = dirname + '_3'
    os.makedirs(spath, exist_ok=True)
    trans_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join(spath, imname), trans_image)
