import QtQuick
import QtQuick.Layouts
import QtQuick.Controls as QQC2
import org.kde.plasma.plasmoid
import org.kde.kirigami as Kirigami

PlasmoidItem {
    id: root

    // idle | listening | thinking | done | error
    property string phase: "idle"
    property string transcript: ""
    property string said: ""
    property string errorMsg: ""
    property bool wakeOn: false
    property bool wakeKnown: false

    function pollWake() {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "http://127.0.0.1:8770/wake");
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE)
                return;
            if (xhr.status === 200) {
                try {
                    root.wakeOn = JSON.parse(xhr.responseText).on === true;
                    root.wakeKnown = true;
                } catch (e) {}
            }
        };
        xhr.send();
    }

    function toggleWake() {
        var want = !root.wakeOn;
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://127.0.0.1:8770/wake");
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE)
                return;
            if (xhr.status === 200) {
                try {
                    root.wakeOn = JSON.parse(xhr.responseText).on === true;
                    root.wakeKnown = true;
                } catch (e) {}
            }
        };
        xhr.send(JSON.stringify({ on: want }));
    }

    Component.onCompleted: pollWake()
    Timer { interval: 10000; running: true; repeat: true; onTriggered: root.pollWake() }

    readonly property color micColor: phase === "error" ? Kirigami.Theme.negativeTextColor
        : (phase === "listening" || phase === "thinking") ? Kirigami.Theme.highlightColor
        : Kirigami.Theme.highlightColor

    Timer {
        id: thinkTimer
        interval: 2000
        onTriggered: if (root.phase === "listening") root.phase = "thinking"
    }

    function listen() {
        if (phase === "listening" || phase === "thinking")
            return;
        transcript = ""; said = ""; errorMsg = "";
        phase = "listening";
        thinkTimer.restart();
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://127.0.0.1:8770/listen");
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE)
                return;
            thinkTimer.stop();
            if (xhr.status === 200) {
                try {
                    var r = JSON.parse(xhr.responseText);
                    root.transcript = r.transcript || "";
                    root.said = r.said || "";
                    root.phase = "done";
                } catch (e) {
                    root.errorMsg = "Bad response";
                    root.phase = "error";
                }
            } else {
                root.errorMsg = xhr.status === 0 ? "Daemon not reachable" : ("HTTP " + xhr.status);
                root.phase = "error";
            }
        };
        xhr.send("");
    }

    function statusText() {
        if (phase === "listening") return i18n("Listening…");
        if (phase === "thinking") return i18n("Thinking…");
        if (phase === "error") return errorMsg;
        if (phase === "done") return said;
        return i18n("Tap to speak");
    }

    // Shared mic UI used by both representations.
    Component {
        id: micUi
        ColumnLayout {
            spacing: Kirigami.Units.largeSpacing
            Item {
                Layout.alignment: Qt.AlignHCenter
                Layout.preferredWidth: Math.min(parent ? parent.width : 160, 160)
                Layout.preferredHeight: Layout.preferredWidth
                Rectangle {
                    id: btn
                    anchors.centerIn: parent
                    width: Math.min(parent.width, parent.height)
                    height: width
                    radius: width / 2
                    color: root.micColor
                    opacity: (root.phase === "listening" || root.phase === "thinking")
                        ? (pulse.running ? 0.6 : 1.0) : (ma.pressed ? 0.7 : 1.0)
                    Behavior on opacity { NumberAnimation { duration: 400 } }
                    SequentialAnimation on opacity {
                        id: pulse
                        running: root.phase === "listening" || root.phase === "thinking"
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.55; duration: 500 }
                        NumberAnimation { to: 1.0; duration: 500 }
                    }
                    Kirigami.Icon {
                        anchors.centerIn: parent
                        width: parent.width * 0.45
                        height: width
                        source: "audio-input-microphone"
                        color: Kirigami.Theme.highlightedTextColor
                        isMask: true
                    }
                    MouseArea {
                        id: ma
                        anchors.fill: parent
                        onClicked: root.listen()
                    }
                }
            }
            QQC2.Label {
                Layout.alignment: Qt.AlignHCenter
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                text: root.statusText()
                color: root.phase === "error" ? Kirigami.Theme.negativeTextColor
                    : Kirigami.Theme.textColor
                font.bold: root.phase === "listening" || root.phase === "thinking"
            }
            QQC2.Label {
                Layout.alignment: Qt.AlignHCenter
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                visible: root.phase === "done" && root.transcript.length > 0
                text: root.transcript.length > 0 ? i18n("“%1”", root.transcript) : ""
                opacity: 0.7
                font.italic: true
            }
            QQC2.Label {
                Layout.alignment: Qt.AlignHCenter
                text: root.wakeKnown ? (root.wakeOn ? i18n("hey linux: on") : i18n("hey linux: off"))
                                     : i18n("hey linux: …")
                color: root.wakeOn ? Kirigami.Theme.positiveTextColor : Kirigami.Theme.disabledTextColor
                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.toggleWake() }
            }
        }
    }

    preferredRepresentation: fullRepresentation

    compactRepresentation: MouseArea {
        onClicked: root.listen()
        Kirigami.Icon {
            anchors.fill: parent
            source: "audio-input-microphone"
            color: root.micColor
            active: root.phase === "listening" || root.phase === "thinking"
        }
    }

    fullRepresentation: Item {
        Layout.minimumWidth: Kirigami.Units.gridUnit * 12
        Layout.minimumHeight: Kirigami.Units.gridUnit * 14
        Loader {
            anchors.centerIn: parent
            width: parent.width - Kirigami.Units.largeSpacing * 2
            sourceComponent: micUi
        }
    }
}
