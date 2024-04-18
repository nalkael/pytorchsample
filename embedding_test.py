import time
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
print('downloaded...')

model_load_start = time.time()
ft = fasttext.load_model('cc.en.300.bin')
model_load_end = time.time()
print('model loaded... in %.2f second' % (model_load_end - model_load_start))

