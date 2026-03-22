# KnightVision ♟️

KnightVision is a computer vision pipeline that converts real-world chess games (captured on video) into PGN (Portable Game Notation) in near real time.

The system detects the board, classifies pieces, and reconstructs game state across frames to produce a valid sequence of moves from raw video input.

---

## Features

- Detects chessboards and pieces from video frames  
- Reconstructs game state over time and outputs valid PGN  
- Uses ONNX models for efficient inference  
- Simple CLI built with Typer  
- Installable via pip  

---

## How it works

1. Detect the chessboard region in each frame  
2. Classify pieces on the board using a trained model  
3. Track changes between frames to infer moves  
4. Convert those moves into standard PGN format  

---

## Example Output

```text
[Event "Casual Game"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ...
```

---

## Quick start

### Install
```bash
pip install knightvision
```

### Download models (one-time setup)
```bash
knightvision models download all
```

### Locate models
```bash
knightvision models locate
```

### Run on a video
```bash
knightvision run --video /path/to/game.mp4 --out /path/to/output.pgn --show
```

---

## Notes

- Performance depends on video quality, lighting, and camera angle  
- Some edge cases (e.g., underpromotion) may not be fully supported  

---

## Tech Stack

- Python  
- OpenCV  
- ONNX Runtime  
- YOLO-based object detection  
- Typer (CLI)
