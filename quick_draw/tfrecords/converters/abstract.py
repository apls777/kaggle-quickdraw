from abc import ABC, abstractmethod


class AbstractConverter(ABC):

    @abstractmethod
    def convert(self, strokes: list):
        """
        :type strokes: strokes in simplified format
        """
        raise NotImplementedError
