#requires pip install gTTS

from gtts import gTTS
#import os

with open('input.txt', 'r', encoding='utf-8') as f:
    content = f.read()

mytext = content
language = 'en'

myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("input.mp3")
#os.system("start input.mp3")
