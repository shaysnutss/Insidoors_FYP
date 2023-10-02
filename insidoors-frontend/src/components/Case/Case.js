import { TicketHighPriority } from "./Ticket/Ticket";
import "./Case.css";

export const Case = () => {
  return (
    <div className="case">
      <div className="div-2">
        <div className="overlap-2">
          <div className="ellipse" />
          <div className="text-wrapper-4">Insidoors</div>
        </div>
        <div className="overlap-3">
          <div className="text-wrapper-5">Dashboard</div>
          <div className="text-wrapper-6">Employees</div>
          <div className="text-wrapper-7">Case Management</div>
        </div>
        <div className="overlap-4">
          <div className="text-wrapper-8">Logout</div>
          <div className="text-wrapper-9">Alerts</div>
        </div>
        <div className="overlap-5">
          <div className="frame">
          <TicketHighPriority />
          </div>
          <div className="text-wrapper-10">Open Cases</div>
          <img className="image-2" alt="Image" src="image-3.png" />
        </div>
        <div className="overlap-6">
          <div className="ticket-high-priority-wrapper">
            <div className="ticket-high-priority-2">
              <img className="line-2" alt="Line" src="line-8-2.svg" />
              <div className="overlap-7">
                <div className="overlap-group-2">
                  <div className="rectangle-2" />
                  <div className="text-wrapper-11">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-12">DESCRIPTION</div>
                  <div className="text-wrapper-13">Title</div>
                  <p className="text-wrapper-14">
                    Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.
                  </p>
                </div>
              </div>
              <img className="image-3" alt="Image" src="image-7-2.png" />
              <img className="image-4" alt="Image" src="image-8-3.png" />
            </div>
          </div>
          <div className="text-wrapper-10">Assigned</div>
          <img className="image-5" alt="Image" src="image-5.png" />
        </div>
        <div className="overlap-8">
          <div className="div-wrapper">
            <div className="ticket-high-priority-2">
              <img className="line-2" alt="Line" src="image.svg" />
              <div className="overlap-7">
                <div className="overlap-group-2">
                  <div className="rectangle-2" />
                  <div className="text-wrapper-11">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-12">DESCRIPTION</div>
                  <div className="text-wrapper-13">Title</div>
                  <p className="text-wrapper-14">
                    Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.
                  </p>
                </div>
              </div>
              <img className="image-3" alt="Image" src="image.png" />
              <img className="image-4" alt="Image" src="image-8-2.png" />
            </div>
          </div>
          <div className="text-wrapper-10">In Review</div>
          <img className="image-6" alt="Image" src="image-4.png" />
        </div>
        <div className="overlap-9">
          <div className="frame-2">
            <div className="ticket-high-priority-2">
              <img className="line-2" alt="Line" src="line-8.svg" />
              <div className="overlap-7">
                <div className="overlap-group-2">
                  <div className="rectangle-2" />
                  <div className="text-wrapper-11">High</div>
                </div>
                <div className="overlap-group-3">
                  <div className="text-wrapper-12">DESCRIPTION</div>
                  <div className="text-wrapper-13">Title</div>
                  <p className="text-wrapper-14">
                    Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.
                  </p>
                </div>
              </div>
              <img className="image-3" alt="Image" src="image-7.png" />
              <img className="image-4" alt="Image" src="image-8.png" />
            </div>
          </div>
          <div className="text-wrapper-10">Closed</div>
          <img className="image-7" alt="Image" src="image-6.png" />
        </div>
        <div className="overlap-10">
          <div className="rectangle-3" />
          <div className="text-wrapper-15">Search</div>
        </div>
      </div>
    </div>
  );
};
export default Case