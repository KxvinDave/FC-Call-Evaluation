class SpeakerIdentification:
    def identification(self, transcript):
        """
        Function to identify speakers in a transcript.
        """
        Speakers = {}
        currentSpeaker = None
        line = transcript.split('\n')
        if line.startswith("SPEAKER"):
            currentSpeaker = line.split()[1]
            Speakers[currentSpeaker] = "Member"
        elif currentSpeaker:
            if "certified" in line.lower() or "concierge" in line.lower():
                Speakers[currentSpeaker] = "FC"
        return Speakers
    
    def replaceTags(self, transcript, speakerRoles):
        lines = transcript.split("\n")
        for i, line in enumerate(lines):
            for speaker, label in speakerRoles.items():
                if line.startswith(speaker):
                    lines[i] = line.replace(speaker, label)
        return "\n".join(lines)