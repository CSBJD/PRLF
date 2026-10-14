# PRLF implementation

Run with the installed environment:

```bash
python run_prlf.py --smoke
python run_prlf.py --dataset mosi --epochs 50
python run_prlf.py --dataset mosi --mode test --checkpoint pt/prlf.pth
python run_prlf.py --dataset mosi --mode test --checkpoint pt/prlf.pth --missing-rate 0 --available av
```

This project contains only PRLF. DMD checkpoints cannot initialize PRLF.
