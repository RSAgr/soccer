# Overview

Through this project we try to assign soccer players with ids which would remain consistent throughout the match. We tried different tracking tracking methods, starting with Deepsort then moving to StrongSort. With both almost a same level of satisfactory performance was been able to be achieved. In this github upload we have kept the Deepsort version, though everyone is welcome to reach out for the StrongSort version.

## Input

Project was tried on the following input:
[Google Drive - Input Video](https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive_link)

## Output

Corresponding Output:
[Google Drive - Output Video](https://drive.google.com/file/d/1hjQxy6GUyPdryagLr2WHQ2k09JbWS4Ak/view?usp=drive_link)

## Note from the Developer

This project is a result of several experimentation with the embedding models, model parameters and various other techniques including box shrinking, color masking (for grass changing -- green color), similarity matching, velocity and position tracking, etc.

Currently we have achieved a great percentage of successful and fast detection of the players , along with a decent level of tracking. Currently, it give unsolicitated results on a few scenarios, especially when two players cross/collide with each other - causing there ids to swap/change. Though we were able to reduce the number of cases by altering tracking model parameters, similarity matching and position tracking. Velocity tracking has also been incorporated but the improvement didn't seem significant, maybe due to its last minute and basic implementation. The velocity tracking still seems to be a potential solution for the id swapping issue. 

## To run the project locally

Install Dependencies:
pip install -r requirements.txt

Download the torchreid library:
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git 

Download model by directly running the following:
bash download_model.sh

OR

Manually download the model from the following link and place it under a folder named models
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view



