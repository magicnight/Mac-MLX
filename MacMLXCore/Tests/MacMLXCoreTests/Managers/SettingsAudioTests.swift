import Testing
import Foundation
@testable import MacMLXCore

@Suite("Settings audio fields (v0.6)")
struct SettingsAudioTests {

    @Test
    func defaultSettingsHaveAudioOffAndNoModelsPicked() {
        let s = Settings.default
        #expect(s.audioEnabled == false)
        #expect(s.sttModel == nil)
        #expect(s.ttsModel == nil)
        #expect(s.ttsVoice == nil)
        #expect(s.ttsAutoSpeak == false)
    }

    @Test
    func roundTripsThroughJSON() throws {
        var s = Settings.default
        s.audioEnabled = true
        s.sttModel = "whisper-medium"
        s.ttsModel = "marvis"
        s.ttsVoice = "voices/clone-kevin.wav"
        s.ttsAutoSpeak = true

        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(Settings.self, from: data)
        #expect(back.audioEnabled == true)
        #expect(back.sttModel == "whisper-medium")
        #expect(back.ttsModel == "marvis")
        #expect(back.ttsVoice == "voices/clone-kevin.wav")
        #expect(back.ttsAutoSpeak == true)
    }

    /// Pre-v0.6 settings.json files don't carry any of the audio
    /// keys — the decoder must default to "audio off" so existing
    /// installs upgrade without surprise.
    @Test
    func legacyJSONWithoutAudioKeysDecodesWithAudioOff() throws {
        let legacy = """
        {
            "modelDirectory": "file:///tmp/models",
            "preferredEngine": "mlx-swift-lm",
            "serverPort": 8000,
            "autoStartServer": false,
            "lastLoadedModel": null,
            "onboardingComplete": true,
            "pythonPath": null,
            "swiftLMPath": null,
            "sparkleUpdateChannel": "release",
            "logRetentionDays": 7
        }
        """
        let decoded = try JSONDecoder().decode(Settings.self, from: Data(legacy.utf8))
        #expect(decoded.audioEnabled == false)
        #expect(decoded.sttModel == nil)
        #expect(decoded.ttsModel == nil)
        #expect(decoded.ttsVoice == nil)
        #expect(decoded.ttsAutoSpeak == false)
    }
}
