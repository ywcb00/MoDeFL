import csv
import os

# static clss for incrementally storing information about performance and saving them in a log file
class PerformanceLogger:
    logdict = dict()

    # append one entry to the specified collection in the logging dictionary
    @classmethod
    def log(self_class, logpath, valuedict):
        self_class.logdict.setdefault(logpath, []).append(valuedict)

    # save the eventual log file to disk
    @classmethod
    def write(self_class):
        for logpath, valuearr in self_class.logdict.items():
            os.makedirs(os.path.dirname(logpath), exist_ok=True)
            with open(f'{logpath}.csv', 'w') as outfile:
                writer = csv.DictWriter(outfile, valuearr[0].keys())
                writer.writeheader()
                writer.writerows(valuearr)
