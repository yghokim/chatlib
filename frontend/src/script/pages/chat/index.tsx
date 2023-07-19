import {IntroView} from "./IntroView";
import {useEffect, useRef, useState} from "react";
import {nanoid} from "nanoid";
import {useSelector} from "../../redux/hooks";
import {ChatView} from "./ChatView";

export const ChatPage = ()=>{

    const [sessionId, setSessionId] = useState<string|undefined>(undefined)

    useEffect(()=>{
        setSessionId(nanoid())
    }, [])

    const isInitialized = useSelector(state => state.chatState.sessionInfo != null)


    return <div>
        {
            isInitialized ? <ChatView/> : <IntroView sessionId={sessionId!!}/>
        }
    </div>
}