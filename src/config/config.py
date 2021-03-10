from pydantic import BaseModel


class Config(BaseModel):
    @property
    def class_name(self):
        return self.__class__.__name__

