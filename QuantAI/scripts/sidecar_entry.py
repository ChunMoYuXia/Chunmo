from multiprocessing import freeze_support

from portfolio_client.__main__ import main

if __name__ == "__main__":
    freeze_support()
    main()
