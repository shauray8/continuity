from __future__ import annotations
import shutil
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, tempfile, pathlib, string, ctypes, sys, gzip, getpass
import urllib.request, subprocess, shutil, math, contextvars, types, copyreg, inspect, importlib
from dataclasses import dataclass
from typing import Union, ClassVar, Optional, Iterable, Any, TypeVar, Callable, Sequence, TypeGuard, Iterator, Generic
from tqdm import tqdm

T = TypeVar("T")

class cqdm(tqdm):
    def __init__(self, *args, **kwargs):
        self.desc = kwargs.get('desc', 'Loading')
        super().__init__(*args, **kwargs)
    
    def format_meter(self, n, total, elapsed, ncols=None, prefix='', **kwargs):
        percentage = n / total if total else 0
        bar_length = 30
        filled_length = int(bar_length * percentage)
        bar = '⌘' * filled_length + "ᓚᘏᗢ"  if filled_length < bar_length else '⌘' * (bar_length+3)
        bar = bar.ljust(bar_length)
        return f"•ﻌ• {self.desc}: [{bar}] {int(percentage * 100)}% •ﻌ•"

class tqdm(Generic[T]):
  def __init__(self, iterable:Iterable[T]|None=None, desc:str='', disable:bool=False,
               unit:str='it', unit_scale=False, total:Optional[int]=None, rate:int=100):
    self.iterable, self.disable, self.unit, self.unit_scale, self.rate = iterable, disable, unit, unit_scale, rate
    self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, getattr(iterable, "__len__", lambda:0)() if total is None else total
    self.set_description(desc)
    self.update(0)
  def __iter__(self) -> Iterator[T]:
    assert self.iterable is not None, "need an iterable to iterate"
    for item in self.iterable:
      yield item
      self.update(1)
    self.update(close=True)
  def __enter__(self): return self
  def __exit__(self, *_): self.update(close=True)
  def set_description(self, desc:str): self.desc = f"{desc}: " if desc else ""
  def update(self, n:int=0, close:bool=False):
    self.n, self.i = self.n+n, self.i+1
    if self.disable or (not close and self.i % self.skip != 0): return
    prog, elapsed, ncols = self.n/self.t if self.t else 0, time.perf_counter()-self.st, shutil.get_terminal_size().columns
    if elapsed and self.i/elapsed > self.rate and self.i: self.skip = max(int(self.i/elapsed)//self.rate, 1)

    def HMS(t): return ':'.join(f'{x:02d}' if i else str(x) for i,x in enumerate([int(t)//3600,int(t)%3600//60,int(t)%60]) if i or x)
    def SI(x): return (f"{x/1000**int(g:=round(math.log(x,1000),6)):.{int(3-3*math.fmod(g,1))}f}"[:4].rstrip('.')+' kMGTPEZY'[int(g)].strip()) if x else '0.00'

    prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
    est_text = f'<{HMS(elapsed/prog-elapsed) if self.n else "?"}' if self.t else ''
    it_text = (SI(self.n/elapsed) if self.unit_scale else f"{self.n/elapsed:5.2f}") if self.n else "?"
    suf = f'{prog_text} [{HMS(elapsed)}{est_text}, {it_text}{self.unit}/s]'

    cat = "ᓚᘏᗢ"
    sz = max(ncols - len(self.desc) - len(suf) - 10, 1)  # reserve 10 extra chars for spacing/safety
    cat_pos = int(sz * prog)
    filled = "⌘" * max(cat_pos - 1, 0)
    space = " " * max(sz - cat_pos, 0)
    bar = '\r' + self.desc + (f'{100*prog:3.0f}%|{filled}{cat}{space}| ' if self.t else '') + suf
    print(bar[:ncols+1], flush=True, end='\n'*close, file=sys.stderr)
  @classmethod
  def write(cls, s:str): print(f"\r\033[K{s}", flush=True, file=sys.stderr)


if __name__ == "__main__":
    for _ in tqdm(range(20), "BENCHMARK RUN"):
        import time
        time.sleep(.1)
