import os
import shutil
import warnings

try:
    import WindowsError
except ImportError:
    WindowsError = None


def _delete_folder(folder_path, warn=False):
    """Utility function to cleanup a temporary folder if still existing.
    Copy from sklearn.utils.testing (for independance)"""
    try:
        if os.path.exists(folder_path):
            # This can fail under windows,
            #  but will succeed when called by atexit
            shutil.rmtree(folder_path)
    except WindowsError:
        if warn:
            warnings.warn("Could not delete temporary folder %s" % folder_path)
