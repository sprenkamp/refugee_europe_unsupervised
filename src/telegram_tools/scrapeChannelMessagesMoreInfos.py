from telethon import TelegramClient, events, sync
import telethon
import pandas as pd
import asyncio
import os
from dotenv import load_dotenv
load_dotenv() 
import datetime
from tqdm import tqdm 
import argparse
import time


# These example values won't work. You must get your own api_id and
# api_hash from https://my.telegram.org, under API Development.

TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID") #add your own api_id
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH") #add your own api_hash

def validate_file(f): #function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

async def callAPI(input_file, output_folder_path, images):
    with open(input_file) as file:
        chats = file.readlines()
        chats = [chat.replace('\n','') for chat in chats if not chat.startswith("#")]
    chatHttps = []
    messageSender = []
    messageID = []
    messageReplyID = []
    messageText = []
    messageDatetime = []
    messageViews=[] 
    messageForwards=[]
    messageReactions = []
    messageMedia = []
    # time = []
    for chat in tqdm(chats):
        print('scrapping chat:', chat)
        async with TelegramClient('testSession', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
            # start = time.time()
            chat_short=chat.split('/')[-1]
            async for message in client.iter_messages(chat):
                if float(message.id)%10000 == 0:
                    print (message.id)
                chatHttps.append(chat)
                messageSender.append(message.sender_id)
                messageID.append(message.id)
                messageReplyID.append(message.reply_to_msg_id)
                messageText.append(message.text)
                messageDatetime.append(message.date)
                messageViews.append(message.views)
                messageForwards.append(message.forwards)
                messageReactions.append(message.reactions)
                if images:
                    if message.photo:
                        try: 
                            path = await client.download_media(message.media, f'{output_folder_path}images/{chat_short}_{message.id}')
                        except telethon.errors.rpcerrorlist.FileMigrateError:
                            "file download not possible"
                        except ValueError:
                            "File download not possible"
            # end = time.time()
            # print(end - start)
    df = pd.DataFrame({'chat':chatHttps, 
                       'messageSender':messageSender, 
                       'messageID':messageID,
                       'messageReplyID':messageReplyID,
                       'messageDatetime':messageDatetime, 
                       'messageViews':messageViews,
                       'messageForwards':messageForwards,
                       'messageReactions':messageReactions,
                       'messageText':messageText})
    df.to_csv(f'{output_folder_path}df.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="Specify the input file", type=validate_file, required=True) #TODO change to argparse.FileType('r')
    parser.add_argument('-o', '--output_folder', help="Specify location of output folder", required=True)
    parser.add_argument('-im', '--images', help="downloads images of groups", action='store_true')
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(callAPI(args.input_file, args.output_folder, args.images))
    loop.close()

if __name__ == '__main__':
    main()