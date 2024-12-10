#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }



# CameraHMR
echo -e "\nYou need to register at https://camerahmr.is.tue.mpg.de/"
read -p "Username (CameraHMR):" username
read -p "Password (CameraHMR):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/training-data
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=insta1-release.npz' -O './data/training-labels/insta1-release.npz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=insta2-release.npz' -O './data/training-labels/insta2-release.npz' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=aic-release.npz' -O './data/training-labels/aic-release.npz' --no-check-certificate --continue
