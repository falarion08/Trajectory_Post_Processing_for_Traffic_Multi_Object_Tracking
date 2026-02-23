class Track:
  def __init__(self,start_frame:int,end_frame:int,track_list:list):
    self.start_frame = start_frame
    self.end_frame = end_frame
    self.track_list = track_list

  def __str__(self):
    track_list_str = "" # Initialize an empty string
    for item in self.track_list:
      track_list_str += f"{item}\n"
    return f"""
Start Frame: {self.start_frame}
End Frame: {self.end_frame}
Total Trajectories: {len(self.track_list)}
Track List:[
{track_list_str}
]
            """