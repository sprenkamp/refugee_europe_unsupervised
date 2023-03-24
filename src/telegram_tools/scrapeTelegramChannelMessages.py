from telethon import TelegramClient, events, sync
import pandas as pd
import asyncio
import os
from dotenv import load_dotenv
load_dotenv() 
import datetime
from tqdm import tqdm 
import argparse
import time


# To run this code. You must get your own api_id and
# api_hash from https://my.telegram.org, under API Development.

TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID") #add your own api_id, we load this from .env file
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH") #add your own api_hash, we load this from .env file

def validate_file(f): #function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

async def callAPI(input_file_path, output_file_path):
    """
    This function takes an input file, output folder path
    It reads the input file, extracts the chats and then uses the TelegramClient to scrape message.text and message.date from each chat.
    Appending the chat's URL, message text, and message datetime to different lists.
    Then it creates a dataframe from the lists and saves the dataframe to a CSV file in the specified output folder.
    
    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    :output_folder_path: folder path where the output CSV file will be saved containing the scraped data
    """

    with open(input_file_path) as file:
        chats = file.readlines()
        chats = [chat.replace('\n','') for chat in chats if not chat.startswith("#")]
    if os.path.isfile(output_file_path):
        df = pd.read_csv(output_file_path)
        df.messageDatetime = pd.to_datetime(df.messageDatetime)
    else:
        df = pd.DataFrame({'chat': [], 
                       'messageDatetime': [], 
                       'messageText': []})
    chatHttps = []
    messageText = []
    messageDatetime = []
    for chat in tqdm(chats):
        if chat in df.chat.unique():
            max_time = df[df.chat == chat].messageDatetime.max()
            print("scraping chat {} since {}".format(chat, max_time))
        else:
            print('scrapping full chat:', chat)
            max_time = None
        async with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
            # chat_short=chat.split('/')[-1]
            async for message in client.iter_messages(chat, reverse = True, offset_date=max_time):
                chatHttps.append(chat)
                messageText.append(message.text)
                messageDatetime.append(message.date)
        df = pd.concat([df, pd.DataFrame({'chat': chatHttps, 
                             'messageDatetime': messageDatetime, 
                             'messageText': messageText})], ignore_index=True)
        df = df.drop_duplicates(['chat', 'messageDatetime', 'messageText'])
        df = df.sort_values(by=['chat', 'messageDatetime'])
        df.to_csv(f'{output_file_path}', index=False)
        print("scraped {} telegram messages".format(len(df)))

def main():
    """
    example usage in command line:
    python src/telegram_tools/scrapeTelegramChannelMessages.py -i data/telegram/queries/DACH.txt -o data/telegram/DACH/df.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="Specify the input file", type=validate_file, required=True)
    parser.add_argument('-o', '--output_file', help="Specify location of output folder", required=True)
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(callAPI(args.input_file, args.output_file))
    loop.close()

if __name__ == '__main__':
    main()