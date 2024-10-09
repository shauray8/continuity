from typing import Optional
import os

class Cache:
    def __init__(self) -> None:
        self.path_to_drectory = None

    def _create_folder(self):
        folder_name = ".continuity"
        user_home = os.path.expanduser("~")
        folder_path = os.path.join(user_home,folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.path_to_drectory = folder_path

    def cached(self):
        if self.path_to_drectory is None :
            self._create_folder()

        ## logcv nees to completd here ..
        
    def get_all_cached_models(self):
        list_of_models = os.listdir(self.path_to_drectory)
        return list_of_models


    def get_model_path(self,name):
        if self.check_model_exists(name):
            pass

    def check_model_exists(self,name: str):
        list_of_models = os.listdir(self.path_to_drectory)

        if name in list_of_models :
            ## check if all the checkpoints are correct
            return True

        else :
            return False

    def cache_model(self,name,version):
        if self.check_model_exists(name=name):
            return 