import os
from viewer import SimpleViewer

if __name__ == "__main__":
    print(os.path.abspath("."))
    viewer = SimpleViewer()
    viewer.run()
