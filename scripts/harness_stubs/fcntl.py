LOCK_EX = 2
LOCK_UN = 8
F_GETFL = 3
F_SETFL = 4


def flock(fileobj, operation):
    del fileobj, operation
    return 0


def fcntl(fd, command, arg=0):
    del fd, command
    return arg
