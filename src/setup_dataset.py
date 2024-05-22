import os
import pathlib
import time
import requests
import tarfile
from zipfile import ZipFile

def GetProjectDir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent

# Print iterations progress
def PrintProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd='\r'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    # Print New Line on Complete
    if iteration == total:
        print()

def DownloadFile(URL, filePath) -> bool:
    if type(URL) is not str:
        print('ERROR: URL should be a str!')
        return False

    fileName = URL.split('/')[-1]

    pathlib.Path(filePath).mkdir(parents=True, exist_ok=True)
    print(f'INFO: Preparing to download: {fileName} to path: {filePath}')

    start = time.perf_counter()

    response = requests.get(URL, stream=True)
    
    # Check if 'content-length' header is present
    fileSizeHeader = response.headers.get('content-length')
    if fileSizeHeader is not None:
        fileSize = int(fileSizeHeader)
    else:
        fileSize = None
        print('WARNING: Unable to determine file size. Progress bar will not be accurate.')

    downloaded = 0

    if fileSize is not None:
        print(f'INFO: Downloading! {fileName} Download Size: {round(fileSize / 1024 / 1024, 2)} Mb')

    with open(f'{filePath}/{fileName}', 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if fileSize is not None:
                    suffix = f'({round(downloaded / 1024 / 1024, 2)} / {round(fileSize / 1024 / 1024, 2)} MB)'
                    try:
                        width = os.get_terminal_size().columns - 50
                        printEnd = '\r'
                    except OSError:
                        width = 75
                        printEnd = ''
                    PrintProgressBar(downloaded, fileSize, length=width, suffix=suffix, printEnd=printEnd)

    if fileSize is not None and fileSize != downloaded:
        print(f'\nERROR: Failed to download {fileName}!')
        return False

    print(f'INFO: Done! The download took {round(time.perf_counter() - start, 1)} seconds!')
    return True




def UnzipFile(fileExt, src, destFolder):
    if fileExt == 'zip':
        with ZipFile(src, 'r') as zipFile:
            # extracting all the files
            print(f'INFO: Extracting {src}')
            zipFile.extractall(destFolder)
    elif fileExt == 'tar.gz':
        file = tarfile.open(src)
        file.extractall(destFolder)
        file.close()
    else:
        print('INFO: ERROR: Compressed file extension is not supported!')
        return
    print(f'INFO: Extracted {src} to {destFolder}')


if __name__ == '__main__':
    datasetDir = GetProjectDir() / 'dataset'

    # Check if dataset is already downloaded
    if os.path.exists(datasetDir) and os.path.getsize(datasetDir) >= pow(2, 30):
        print('WARNING: SS594_Multispectral_Dehazing already present!')
        exit()

    url = 'https://vedas.sac.gov.in/static/pdf/SIH_2022/SS594_Multispectral_Dehazing.zip'
    print(f'INFO: SS594_Multispectral_Dehazing.zip will be downloaded from the URL: {url}')

    downloadPath = datasetDir
    filename = str(url.split('/')[-1])
    extension = str(filename.split('.')[-1])

    success = DownloadFile(url, str(downloadPath.resolve()))
    downloadedFilePath = downloadPath / f'{filename}'

    if not success:
        if os.path.exists(datasetDir / filename):
            os.remove(datasetDir / filename)
        exit()

    pure_filename = filename.replace(f'.{extension}', '')

    print(f"INFO: Extracting {downloadedFilePath}...")
    UnzipFile(extension, str(downloadedFilePath.resolve()), str(downloadPath.resolve()))

    if os.path.exists(downloadedFilePath):
        os.remove(downloadedFilePath)