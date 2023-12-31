"""
In this example, the number of workers is 3, while the number of values is 14 (just a number)...
The sleep_value of 1 sec is for the observer to see that in each cycle only 3 jobs are carries out in parallel.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from time import sleep

start_time = time.time()
values = list(range(14))
sleep_value = 1


def task(n):
    start_time_task = time.time()
    print(f"{n} start task {n} at time {time.time() - start_time:.4f}\n")
    sleep(sleep_value)
    print(f"{n} end task {n} at time {time.time() - start_time:.4f}. Took {time.time() - start_time_task:.4f}\n")


def main():
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(task, values)


if __name__ == '__main__':
    main()
