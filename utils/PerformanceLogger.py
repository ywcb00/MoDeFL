import csv
import os

class PerformanceLogger:
    logdict = dict()

    @classmethod
    def log(self_class, logpath, valuedict):
        self_class.logdict.setdefault(logpath, []).append(valuedict)

    @classmethod
    def write(self_class):
        for logpath, valuearr in self_class.logdict.items():
            os.makedirs(os.path.dirname(logpath), exist_ok=True)
            with open(f'{logpath}.csv', 'w') as outfile:
                writer = csv.DictWriter(outfile, valuearr[0].keys())
                writer.writeheader()
                writer.writerows(valuearr)
