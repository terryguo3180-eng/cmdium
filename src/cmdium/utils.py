from enum import Enum


class DataEnum(Enum):
  def __repr__(self):
    return f"{type(self).__name__}.{self.name}"

