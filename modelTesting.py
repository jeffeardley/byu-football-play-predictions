import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from THEMODEL import THEMODEL
from readData import readData


def main():
    model = THEMODEL()
    dataFrames = readData()
    model.train(dataFrames)


if __name__ == '__main__':
    main()