# face-attendance-system

This project is a face recognition–based attendance system built using Python and deep learning. 
It allows users to register their face and mark attendance using a live webcam. Attendance is saved automatically with name, time, and action (IN/OUT).

The system uses MTCNN for face detection and FaceNet (InceptionResnetV1) for generating facial embeddings. 
These embeddings are stored locally and compared in real time to recognize users.

Users can register their face, punch in, punch out, and view attendance records through a simple Streamlit interface. 
The system works best in good lighting and provides basic spoof prevention using live camera input.

This project is suitable for learning face recognition and can be extended with features like blink detection, database storage, or cloud deployment.

