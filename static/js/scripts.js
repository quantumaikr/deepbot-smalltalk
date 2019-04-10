window.onload = function() {

    // 메시지 변수 생성
    var message = ""
    var messagetmp = ""

    var synthesis = window.speechSynthesis

    // 설명
    var utterance = new SpeechSynthesisUtterance()

    utterance.text = message
    utterance.lang = 'ko-KR' // 언어 지정; 영어 -> 'en-US'
    // utterance.lang = 'en-US' // 언어 지정; 영어 -> 'en-US'
    utterance.volume = "1" // 소리 크기 값; 최소값: 0, 최댓값: 1
    utterance.pitch = "1" // 음높이, 음의 고저의 정도; 최솟값: 0, 최댓값: 2
    utterance.rate = "1" // 속도; 최솟값: 0.1, 최댓값: 10

    // 음성 인식 객체 생성
    var recognition = new(window.SpeechRecognition || window.webkitSpeechRecognition || window.mozSpeechRecognition || window.msSpeechRecognition)()

    recognition.lang = 'ko-KR' // 언어 지정;
    // recognition.lang = 'en-US' // 언어 지정; 영어 -> 'en-US' 영어 -> 'en-US'
    recognition.interimResults = false // 음성인식 중간에 결과를 반환할지 여부
    recognition.continuous = false // 음성인식에 대해 연속 결과를 반환할지 여부
    recognition.maxAlternatives = 1 // 음성인식 결과 최대 수; 기본 값: 1

    // 음성 인식 서비스 결과가 반환 시 실행되는 이벤트
    recognition.onresult = function(event) {
        var current = event.resultIndex
        var transcript = event.results[current][0].transcript

        // 음성의 반복되는 버그를 잡기 위함 (수정하지 않는 것을 추천)
        var repeatBug = (current == 1 && transcript == event.results[0][0].transcript)
        if (!repeatBug) {
            messagetmp += transcript
        }

        submitMessage(messagetmp)
    }

    utterance.onend = function() {
      console.log('utterance.onend')

      messagetmp = ''
      recognition.start()
    }
    // 음성인식 서비스가 끊어졌을 때의 이벤트
    recognition.onend = function() {
        if(messagetmp ==="") {
          $('#mic').removeClass('active')
        }
    }

    function submitMessage(msg) {
      // 메시지의 내용을 파이썬이 처리 할 수 있게 보내고 응답을 기다림
      $('.history').prepend('<p> ME: ' + msg + '</p>')

      $.ajax({
          url: '/chatting',
          type: 'POST',
          dataType: 'json',
          contentType: 'application/json',
          data: JSON.stringify({message: msg}),
          success: function(response) {
              message = response.message

              // ()로 묶어서 전달되는 덧붙임 정보는 제거; 사유는 말해야 할 말이 너무 많아지기 때문
              message = message.replace(/\([^\(\)]+\)/gi, '')
              utterance.text = message

              // 타이핑 효과 함수에 메시지를 파라미터로 넘김
              consoleText([message], 'ai_message', ['tomato'])
              synthesis.speak(utterance) // utterance의 text를 음성으로 읽음

              $('.history').prepend('<p> BOT: ' + message + '</p>')
          }
      })
    }
    // 타이핑 효과
    function consoleText(words, id, colors) {

        if (colors === undefined) {
            colors = ['#fff']
        }

        var letterCount = 1
        var x = 1
        var waiting = false
        var target = document.getElementById(id)

        target.setAttribute('style', 'color:' + colors[0])
        target.innerHTML = ''

        var typedText = window.setInterval(function() {

            if (words[0].length + 1 === letterCount) {
                clearInterval(typedText)
            }

            if (letterCount === 0 && waiting === false) {
                waiting = true
                target.innerHTML = words[0].substring(0, letterCount)
            } else if (waiting === false) {
                target.innerHTML = words[0].substring(0, letterCount)
                letterCount += x
            }
        }, 120)
    }

    $("#user_message").keydown(function(key) {
      if (key.keyCode == 13) {
        messagetmp = $('input#user_message').val()
        $('input#user_message').val('')
        // 메시지의 내용을 파이썬이 처리 할 수 있게 보내고 응답을 기다림
        submitMessage(messagetmp)
        return
      }
    })

    $('#mic').on('click', function(event) {
        var flag = $('#mic').hasClass('active')

        if(flag) {
          $('#mic').removeClass('active')
        } else {
          $('#mic').addClass('active')

          messagetmp = ''
          if (messagetmp.length) {
              messagetmp += ' '
          }
          recognition.start()
        }
    })
}
