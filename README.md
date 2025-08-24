# KnightVision ♟️

KnightVision is a computer vision tool that converts **real-world chess games** (recorded from video) into **PGN** (Portable Game Notation) in real time.  It’s designed to showcase applied machine learning and computer vision, with a CLI interface for developers and chess enthusiasts.

---

## Features

-  Detects chessboards and pieces from video input  
-  Tracks game state and outputs valid **PGN**  
-  Supports ONNX models for fast inference  
-  Simple CLI powered by [Typer](https://typer.tiangolo.com/)  
- Installable via `pip` 

---

## Quick start


1) Install
```bash
pip install knightvision
````
2) One-time: download models (from the latest GitHub release)
```bash
knightvision models download all
````
3) See where models are resolved from
```bash
4) knightvision models locate
````
4) Run on a video and show
```bash
5) knightvision run --video /path/to/game.mp4 --out /path/to/out.pgn --show
```