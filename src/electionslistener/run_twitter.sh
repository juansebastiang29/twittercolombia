until python /home/juan/Desktop/Computational_Machine_Learning/twitter_webS.py; do
    echo "Server 'twitter' crashed with exit code $?.  Respawning.." >&2
    sleep 10s
done
