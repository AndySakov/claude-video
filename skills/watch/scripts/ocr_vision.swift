// Apple Vision OCR for watch (macOS only). Reads image paths from stdin, one per
// line, and prints one JSON object per image:
//   {"path": "...", "lines": [{"text": "...", "conf": 0.98, "y": 0.81}]}
// "y" is the line's vertical centre from the top of the image (0-1).
import Foundation
import Vision

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
if let langs = ProcessInfo.processInfo.environment["WATCH_OCR_LANG"], !langs.isEmpty, langs != "eng" {
    request.recognitionLanguages = langs.split(separator: "+").map(String.init)
}

while let raw = readLine() {
    let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.isEmpty { continue }
    var lines: [[String: Any]] = []
    let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
    do {
        try handler.perform([request])
        for case let observation as VNRecognizedTextObservation in request.results ?? [] {
            guard let best = observation.topCandidates(1).first else { continue }
            lines.append([
                "text": best.string,
                "conf": Double(best.confidence),
                "y": Double(1.0 - observation.boundingBox.midY),
            ])
        }
    } catch {
        FileHandle.standardError.write("vision: \(path): \(error)\n".data(using: .utf8)!)
    }
    if let data = try? JSONSerialization.data(withJSONObject: ["path": path, "lines": lines]),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    }
}
