from tqdm import tqdm

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

