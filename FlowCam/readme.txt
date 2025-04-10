Other repository related to FlowCam
LabelChecker (Source code of the program) : https://github.com/TimWalles/LabelChecker
LabelChecker preprocessing (For preprocessing of raw data so it can be access by LabelChecker program) : https://github.com/TimWalles/LabelChecker_Pipeline
- In the branch biovol is the integration of biovolume function into LabelChecker preprocessing scipt. (This will make the data unreadable to LC program)

How to make LabelChecker preprocessing with biovol be readable by LC program

1. In the source code of LabelChecker program

Add
>>  public double BiovolumeHSosik { get; set; }
>>  public double SurfaceAreaHSosik { get; set; }
Into LabelChecker-1.0.2\app\IO\CSVFlowCamFile.cs

Add
>>  "BiovolumeHSosik", "SurfaceAreaHSosik"
Into line 40 of app\Controls\MainMenu\MainMenu.cs

2. Rebuild the program