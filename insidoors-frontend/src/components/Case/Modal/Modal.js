import "./Modal.css";
import React, { useState, useEffect } from "react";
import userService from "../../../services/user.service"
import { line } from "../../../assets"
import axios from "axios";

function Modal({ setOpenModal, caseId }) {

    const [comments, setComments] = useState([]);
    const [detail, setDetail] = useState([]);
    const [socList, setSocList] = useState([]);
    const [tab, setTab] = useState("modal1");
    const [soc, setSoc] = useState("");
    const [socId, setSocId] = useState();
    const [priority, setPriority] = useState("");
    const [name, setName] = useState("");
    const [desc, setDesc] = useState("");


    const fetchComments = async (e) => {
        //e.preventDefault();
        try {
            //const { data } = await userService.getAllCases();
            const { data } = await userService.getAllCommentsById(caseId);
            const comments = data;
            setComments(comments);
        } catch (err) {
            console.log(err);
        }
        console.log(comments);
    };

    const fetchCase = async (e) => {
        //e.preventDefault();
        try {
            const { data } = await userService.getCaseById(caseId);
            const detail = data;
            setDetail(detail);
            setSoc(detail.socName);
            if (detail.severity <= 50) {
                setPriority("Low");
            } else if (detail.severity > 50 && detail.severity <= 100) {
                setPriority("Med");
            } else if (detail.severity > 100) {
                setPriority("High");
            }
        } catch (err) {
            console.log(err);
        }
        console.log(detail);
    };

    const fetchSoc = async (e) => {
        //e.preventDefault();
        try {
            const { data } = await userService.getAllSoc();
            const socList = data;
            setSocList(socList);
        } catch (err) {
            console.log(err);
        }
        console.log(socList);
    };

    const handleModal = async (e) => {
        //e.preventDefault();

        setOpenModal(false);
    };

    const handleSoc = async (e) => {
        //e.preventDefault();
        setSoc(e.target.value);
        setSocId(e.target.key);
        console.log(soc);
        const res = await userService.putSoc(caseId, socId);
    };

    const handleStatus = async (e) => {
        //e.preventDefault();
        const res = await userService.putStatus(caseId, e.target.value);
    };

    useEffect(() => {
        fetchComments()
        fetchCase()
        fetchSoc()
    }, [])

    return (
        <div className="background">
            <div>
                <div className={tab}>
                    <button className="closeButton" onClick={() => { setOpenModal(false); }}> x </button>
                    <img className="lineTop" alt="" src={line} />
                    <div className="textTitle">{detail.incidentTitle}</div>
                    <div className="textEmployee">Employee</div>
                    <div className="textIncident">Incident Timestamp</div>
                    <div className="textSeverity">Severity</div>
                    <div className="textStatus">Status</div>
                    <div className="textSoc">SOC</div>
                    <div className="textDate">Date Assigned</div>
                    <div className="employee">{detail.employeeFirstname} {detail.employeeLastname}</div>
                    <div className="incident">{detail.incidentTimestamp}</div>
                    <div className="severity">{priority}</div>
                    <select className="status" value={detail.status} onChange={handleStatus}>
                        <option value="Open" onChange={handleStatus}>Open</option>
                        <option value="Assigned" onChange={handleStatus}>Assigned</option>
                        <option value="In review" onChange={handleStatus}>In review</option>
                        <option value="Closed" onChange={handleStatus}>Closed</option>
                    </select>
                    <select className="soc" value={soc} onChange={handleSoc}>
                        {socList.map((socList) => (
                            <option key={socList.id} value={socList.socName}>{socList.socName}</option>
                        ))}
                    </select>
                    <div className="date">{detail.dateAssigned}</div>
                    <div className="tab">
                        <button className="descSelected" onClick={() => { setTab("modal1"); }}>DESCRIPTION</button>
                        <button className="commentSelected" onClick={() => { setTab("modal"); }}>COMMENTS</button>
                        <img className="lineBot" alt="" src={line} />
                    </div>
                    <div>
                        {tab === "modal" &&
                            <div className="comments">
                                <div className="commentsGet">
                                    <div>
                                        {comments.map((comments) => (
                                            <div className="box-size" key={comments.id}>
                                                {/* cases.status === "Open" && */
                                                    <div className="commentBox">
                                                        <div className="commentTitle">{comments.socName}</div>
                                                        <div className="commentDescription">
                                                            {comments.commentDescription}
                                                        </div>
                                                        <div className="rectangle" />
                                                    </div>
                                                }
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="addCommentBox">
                                    <input type="text" className="addCommentTitle" placeholder="Enter Name" value={name} onChange={(e) => setName(e.target.value)} />
                                    <input type="text" className="addCommentDescription" placeholder="Enter Description" value={desc} onChange={(e) => setDesc(e.target.value)} />
                                </div>
                            </div>
                        }
                    </div>
                    <div>
                        {tab === "modal1" &&
                            <div className="descBox">{detail.incidentDesc}</div>
                        }
                    </div>
                    <button className="saveButton" onClick={handleModal}>
                        <div className="saveButtonText">Save</div>
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Modal;
