import "./Case.css"
import { openCases, assigned, inReview, closed, plus, message, line } from "../../assets"

const Case = () => {
  return (
    <div className="case">
      <div className="div">
        <div className="overlap">
          <div className="ellipse" />
          <div className="text-wrapper">Insidoors</div>
        </div>
        <div className="overlap-group">
          <div className="text-wrapper-2">Logout</div>
          <div className="text-wrapper-3">Alerts</div>
        </div>
        <div className="overlap-2">
          <div className="text-wrapper-4">Employees</div>
          <div className="text-wrapper-5">Case Management</div>
          <div className="text-wrapper-3">Dashboard</div>
        </div>
        <div className="overlap-3">
          <div className="frame">
            <div className="ticket-high-priority">
              <img className="line" alt="Line" src={line} />
              <div className="overlap-4">
                <div className="overlap-group-2">
                  <div className="rectangle" />
                  <div className="text-wrapper-6">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-8">Title</div>
                  <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                </div>
              </div>
              <img className="image" alt="Image" src={message} />
              <img className="img" alt="Image" src={plus} />
            </div>
          </div>
          <div className="text-wrapper-9">Open Cases</div>
          <img className="image-2" alt="Image" src={openCases} />
        </div>
        <div className="overlap-5">
          <div className="ticket-high-priority-wrapper">
            <div className="ticket-high-priority">
              <img className="line" alt="Line" src={line} />
              <div className="overlap-4">
                <div className="overlap-group-2">
                  <div className="rectangle" />
                  <div className="text-wrapper-6">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-8">Title</div>
                  <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                </div>
              </div>
              <img className="image" alt="Image" src={message} />
              <img className="img" alt="Image" src={plus} />
            </div>
          </div>
          <div className="text-wrapper-9">Assigned</div>
          <img className="image-3" alt="Image" src={assigned} />
        </div>
        <div className="overlap-6">
          <div className="div-wrapper">
            <div className="ticket-high-priority">
              <img className="line" alt="Line" src={line} />
              <div className="overlap-4">
                <div className="overlap-group-2">
                  <div className="rectangle" />
                  <div className="text-wrapper-6">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-8">Title</div>
                  <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                </div>
              </div>
              <img className="image" alt="Image" src={message} />
              <img className="img" alt="Image" src={plus} />
            </div>
          </div>
          <div className="text-wrapper-9">In Review</div>
          <img className="image-4" alt="Image" src={inReview} />
        </div>
        <div className="overlap-7">
          <div className="frame-2">
            <div className="ticket-high-priority">
              <img className="line" alt="Line" src={line} />
              <div className="overlap-4">
                <div className="overlap-group-2">
                  <div className="rectangle" />
                  <div className="text-wrapper-6">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-8">Title</div>
                  <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
                </div>
              </div>
              <img className="image" alt="Image" src={message} />
              <img className="img" alt="Image" src={plus} />
            </div>
          </div>
          <div className="text-wrapper-9">Closed</div>
          <img className="image-5" alt="Image" src={closed} />
        </div>
        <div className="overlap-8">
          <div>
            <input type="text" className="rectangle-2" placeholder="Search" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Case
