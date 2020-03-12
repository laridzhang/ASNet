import progressbar
import sys


class Bar():
    def __init__(self, title='computing', max_value=100):
        self.count = 0
        widgets = [title, ': ', progressbar.Percentage(), ' ', progressbar.Bar(marker='#', left='[', right=']'), ' ', progressbar.ETA(), ]
        self.bar = progressbar.ProgressBar(widgets=widgets, max_value=max_value).start()

    def update(self):
        self.count += 1
        self.bar.update(self.count)
        sys.stdout.flush()

    def done(self):
        self.bar.finish()
        sys.stdout.flush()