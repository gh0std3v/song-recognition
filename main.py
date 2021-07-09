import os
import ShaWOW


if __name__ == '__main__':
    if os.path.exists('database.pkl'):
        ShaWOW.load_database()
    else:
        song_paths = ["song_library/"+ShaWOW.name_to_file[name] for name in ShaWOW.song_names] 
        ShaWOW.fill_database(song_paths, ShaWOW.fingerprint_database, ShaWOW.song_database)
    recorded_audio, sampling_rate = ShaWOW.listen_audio()
    sample_fingerprints = ShaWOW.compute_fingerprint(recorded_audio, sampling_rate)
    result = ShaWOW.compare(sample_fingerprints, ShaWOW.fingerprint_database)
    print("{} by {}".format(ShaWOW.song_names[result], ShaWOW.name_to_artist[ShaWOW.song_names[result]]))
    if not os.path.exists('database.pkl'):
        ShaWOW.create_database()
