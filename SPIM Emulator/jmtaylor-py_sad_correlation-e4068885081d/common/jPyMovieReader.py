import numpy
import jmovies
    
class JPyMovieReader:
    def __init__(self, windowOrigin = (-1, -1), filename = ""):
        self.cpp_obj = jmovies.create_reader(windowOrigin, filename)

    def __del__(self):
        jmovies.destroy_reader(self.cpp_obj)

    def __getitem__(self, frameNo):
        if frameNo < 0 or frameNo >= len(self):
            raise error, 'Seeking to far'
        return jmovies.get_frame_as_array(cpp_obj, frameNo)

    def __len__(self):
        return jmovies.num_frames(cpp_obj)

    def next_frame(self):
        return jmovies.get_next_frame_as_array(cpp_obj)

if __name__ == '__main__':
    # Example
    # Video file converted to uncompressed RGB by VirtualDUB
    reader = JPyMovieReader((10,10), "temp.mov")
    while reader.next_frame() != None:
        print "drew a frame"
